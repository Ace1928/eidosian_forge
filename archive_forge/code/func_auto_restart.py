import os.path
import sys
import yaml
import time
import logging
from argh import arg, aliases, ArghParser, expects_obj
from wandb_watchdog.version import VERSION_STRING
from wandb_watchdog.utils import load_class
@arg('command', help='Long-running command to run in a subprocess.\n')
@arg('command_args', metavar='arg', nargs='*', help='Command arguments.\n\nNote: Use -- before the command arguments, otherwise watchmedo will\ntry to interpret them.\n')
@arg('-d', '--directory', dest='directories', metavar='directory', action='append', help='Directory to watch. Use another -d or --directory option for each directory.')
@arg('-p', '--pattern', '--patterns', dest='patterns', default='*', help='matches event paths with these patterns (separated by ;).')
@arg('-i', '--ignore-pattern', '--ignore-patterns', dest='ignore_patterns', default='', help='ignores event paths with these patterns (separated by ;).')
@arg('-D', '--ignore-directories', dest='ignore_directories', default=False, help='ignores events for directories')
@arg('-R', '--recursive', dest='recursive', default=False, help='monitors the directories recursively')
@arg('--interval', '--timeout', dest='timeout', default=1.0, help='use this as the polling interval/blocking timeout')
@arg('--signal', dest='signal', default='SIGINT', help='stop the subprocess with this signal (default SIGINT)')
@arg('--kill-after', dest='kill_after', default=10.0, help='when stopping, kill the subprocess after the specified timeout (default 10)')
@expects_obj
def auto_restart(args):
    """
    Subcommand to start a long-running subprocess and restart it
    on matched events.

    :param args:
        Command line argument options.
    """
    from watchdog.observers import Observer
    from watchdog.tricks import AutoRestartTrick
    import signal
    import re
    if not args.directories:
        args.directories = ['.']
    if re.match('^SIG[A-Z]+$', args.signal):
        stop_signal = getattr(signal, args.signal)
    else:
        stop_signal = int(args.signal)

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, handle_sigterm)
    patterns, ignore_patterns = parse_patterns(args.patterns, args.ignore_patterns)
    command = [args.command]
    command.extend(args.command_args)
    handler = AutoRestartTrick(command=command, patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=args.ignore_directories, stop_signal=stop_signal, kill_after=args.kill_after)
    handler.start()
    observer = Observer(timeout=args.timeout)
    observe_with(observer, handler, args.directories, args.recursive)
    handler.stop()
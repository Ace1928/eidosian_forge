import os.path
import sys
import yaml
import time
import logging
from argh import arg, aliases, ArghParser, expects_obj
from wandb_watchdog.version import VERSION_STRING
from wandb_watchdog.utils import load_class
@arg('directories', nargs='*', default='.', help='directories to watch')
@arg('-c', '--command', dest='command', default=None, help='shell command executed in response to matching events.\nThese interpolation variables are available to your command string::\n\n    ${watch_src_path}    - event source path;\n    ${watch_dest_path}   - event destination path (for moved events);\n    ${watch_event_type}  - event type;\n    ${watch_object}      - ``file`` or ``directory``\n\nNote::\n    Please ensure you do not use double quotes (") to quote\n    your command string. That will force your shell to\n    interpolate before the command is processed by this\n    subcommand.\n\nExample option usage::\n\n    --command=\'echo "${watch_src_path}"\'\n')
@arg('-p', '--pattern', '--patterns', dest='patterns', default='*', help='matches event paths with these patterns (separated by ;).')
@arg('-i', '--ignore-pattern', '--ignore-patterns', dest='ignore_patterns', default='', help='ignores event paths with these patterns (separated by ;).')
@arg('-D', '--ignore-directories', dest='ignore_directories', default=False, help='ignores events for directories')
@arg('-R', '--recursive', dest='recursive', default=False, help='monitors the directories recursively')
@arg('--interval', '--timeout', dest='timeout', default=1.0, help='use this as the polling interval/blocking timeout')
@arg('-w', '--wait', dest='wait_for_process', action='store_true', default=False, help='wait for process to finish to avoid multiple simultaneous instances')
@arg('-W', '--drop', dest='drop_during_process', action='store_true', default=False, help='Ignore events that occur while command is still being executed to avoid multiple simultaneous instances')
@arg('--debug-force-polling', default=False, help='[debug] forces polling')
@expects_obj
def shell_command(args):
    """
    Subcommand to execute shell commands in response to file system events.

    :param args:
        Command line argument options.
    """
    from watchdog.tricks import ShellCommandTrick
    if not args.command:
        args.command = None
    if args.debug_force_polling:
        from watchdog.observers.polling import PollingObserver as Observer
    else:
        from watchdog.observers import Observer
    patterns, ignore_patterns = parse_patterns(args.patterns, args.ignore_patterns)
    handler = ShellCommandTrick(shell_command=args.command, patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=args.ignore_directories, wait_for_process=args.wait_for_process, drop_during_process=args.drop_during_process)
    observer = Observer(timeout=args.timeout)
    observe_with(observer, handler, args.directories, args.recursive)
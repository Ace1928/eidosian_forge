import sys
import ovs.util
def command_register(name, usage, min_args, max_args, callback, aux):
    """ Registers a command with the given 'name' to be exposed by the
    UnixctlServer. 'usage' describes the arguments to the command; it is used
    only for presentation to the user in "help" output.

    'callback' is called when the command is received.  It is passed a
    UnixctlConnection object, the list of arguments as unicode strings, and
    'aux'.  Normally 'callback' should reply by calling
    UnixctlConnection.reply() or UnixctlConnection.reply_error() before it
    returns, but if the command cannot be handled immediately, then it can
    defer the reply until later.  A given connection can only process a single
    request at a time, so a reply must be made eventually to avoid blocking
    that connection."""
    assert isinstance(name, str)
    assert isinstance(usage, str)
    assert isinstance(min_args, int)
    assert isinstance(max_args, int)
    assert callable(callback)
    if name not in commands:
        commands[name] = _UnixctlCommand(usage, min_args, max_args, callback, aux)
import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
def create_command_class(name, func_module):
    """Dynamically creates subclass of MigratingCommand.

    Method takes name of the function, module it is part of
    and builds the subclass of :py:class:`MigratingCommand`.
    Having a subclass of :py:class:`cliff.command.Command` is mandatory
    for the osc-lib integration.

    :param name: name of the function
    :type name: basestring
    :param func_module: the module function is part of
    :type func_module: module
    :return: command name, subclass of :py:class:`MigratingCommand`
    :rtype: tuple(basestring, class)

    """
    cmd_name = name[3:].replace('_', '-')
    callback = getattr(func_module, name)
    desc = callback.__doc__ or ''
    help = desc.strip().split('\n')[0]
    arguments = getattr(callback, 'arguments', [])
    body = {'_args': arguments, '_callback': staticmethod(callback), '_description': desc, '_epilog': desc, '_help': help}
    claz = type('%sCommand' % cmd_name.title().replace('-', ''), (MigratingCommand,), body)
    return (cmd_name, claz)
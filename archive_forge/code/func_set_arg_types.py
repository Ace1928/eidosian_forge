import functools
import inspect
import logging
import traceback
import wsme.exc
import wsme.types
from wsme import utils
def set_arg_types(self, argspec, arg_types):
    args = argspec.args
    defaults = argspec.defaults
    if args[0] == 'self':
        args = args[1:]
    arg_types = list(arg_types)
    if self.body_type is not None:
        arg_types.append(self.body_type)
    for i, argname in enumerate(args):
        datatype = arg_types[i]
        mandatory = defaults is None or i < len(args) - len(defaults)
        default = None
        if not mandatory:
            default = defaults[i - (len(args) - len(defaults))]
        if datatype is wsme.types.HostRequest:
            self.pass_request = argname
        else:
            self.arguments.append(FunctionArgument(argname, datatype, mandatory, default))
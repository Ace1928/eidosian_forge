import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def map_args(self, args, kwargs, context, engine):
    kwargs = dict(kwargs)
    positional_args = len(args) * [self.parameters.get('*', utils.NO_VALUE)]
    max_dst_positional_args = len(args) + len(self.parameters)
    positional_fix_table = max_dst_positional_args * [0]
    keyword_args = {}
    for p in self.parameters.values():
        if p.position is not None and isinstance(p.value_type, yaqltypes.HiddenParameterType):
            for index in range(p.position + 1, len(positional_fix_table)):
                positional_fix_table[index] += 1
    for key, p in self.parameters.items():
        arg_name = p.alias or p.name
        if p.position is not None and key != '*':
            arg_position = p.position - positional_fix_table[p.position]
            if isinstance(p.value_type, yaqltypes.HiddenParameterType):
                continue
            elif arg_position < len(args) and args[arg_position] is not utils.NO_VALUE:
                if arg_name in kwargs:
                    return None
                positional_args[arg_position] = p
            elif arg_name in kwargs:
                keyword_args[arg_name] = p
                del kwargs[arg_name]
            elif p.default is NO_DEFAULT:
                return None
            elif arg_position < len(args) and args[arg_position]:
                positional_args[arg_position] = p
        elif p.position is None and key != '**':
            if isinstance(p.value_type, yaqltypes.HiddenParameterType):
                continue
            elif arg_name in kwargs:
                keyword_args[arg_name] = p
                del kwargs[arg_name]
            elif p.default is NO_DEFAULT:
                return None
    if len(kwargs) > 0:
        if '**' in self.parameters:
            argdef = self.parameters['**']
            for key in kwargs:
                keyword_args[key] = argdef
        else:
            return None
    for i in range(len(positional_args)):
        if positional_args[i] is utils.NO_VALUE:
            return None
        value = args[i]
        if value is utils.NO_VALUE:
            value = positional_args[i].default
        if not positional_args[i].value_type.check(value, context, engine):
            return None
    for kwd in kwargs:
        if not keyword_args[kwd].value_type.check(kwargs[kwd], context, engine):
            return None
    return (tuple(positional_args), keyword_args)
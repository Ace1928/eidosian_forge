from hashlib import sha1
from ..util import compat
from ..util import langhelpers
def kwarg_function_key_generator(namespace, fn, to_str=str):
    """Return a function that generates a string
    key, based on a given function as well as
    arguments to the returned function itself.

    For kwargs passed in, we will build a dict of
    all argname (key) argvalue (values) including
    default args from the argspec and then
    alphabetize the list before generating the
    key.

    .. versionadded:: 0.6.2

    .. seealso::

        :func:`.function_key_generator` - default key generation function

    """
    if namespace is None:
        namespace = '%s:%s' % (fn.__module__, fn.__name__)
    else:
        namespace = '%s:%s|%s' % (fn.__module__, fn.__name__, namespace)
    argspec = compat.inspect_getargspec(fn)
    default_list = list(argspec.defaults or [])
    default_list.reverse()
    args_with_defaults = dict(((argspec.args[idx * -1], default) for idx, default in enumerate(default_list, 1)))
    if argspec.args and argspec.args[0] in ('self', 'cls'):
        arg_index_start = 1
    else:
        arg_index_start = 0

    def generate_key(*args, **kwargs):
        as_kwargs = dict([(argspec.args[idx], arg) for idx, arg in enumerate(args[arg_index_start:], arg_index_start)])
        as_kwargs.update(kwargs)
        for arg, val in args_with_defaults.items():
            if arg not in as_kwargs:
                as_kwargs[arg] = val
        argument_values = [as_kwargs[key] for key in sorted(as_kwargs.keys())]
        return namespace + '|' + ' '.join(map(to_str, argument_values))
    return generate_key
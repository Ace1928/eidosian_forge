import functools
import inspect
import platform
import warnings
import wrapt
def get_deprecated_msg(self, wrapped, instance):
    """
        Get the deprecation warning message for the user.

        :param wrapped: Wrapped class or function.

        :param instance: The object to which the wrapped function was bound when it was called.

        :return: The warning message.
        """
    if instance is None:
        if inspect.isclass(wrapped):
            fmt = 'Call to deprecated class {name}.'
        else:
            fmt = 'Call to deprecated function (or staticmethod) {name}.'
    elif inspect.isclass(instance):
        fmt = 'Call to deprecated class method {name}.'
    else:
        fmt = 'Call to deprecated method {name}.'
    if self.reason:
        fmt += ' ({reason})'
    if self.version:
        fmt += ' -- Deprecated since version {version}.'
    return fmt.format(name=wrapped.__name__, reason=self.reason or '', version=self.version or '')
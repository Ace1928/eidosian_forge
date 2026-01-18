import inspect
import os
import sys
import traceback
import types
import tensorflow.compat.v2 as tf
def inject_argument_info_in_traceback(fn, object_name=None):
    """Add information about call argument values to an error message.

    Arguments:
      fn: Function to wrap. Exceptions raised by the this function will be
        re-raised with additional information added to the error message,
        displaying the values of the different arguments that the function
        was called with.
      object_name: String, display name of the class/function being called,
        e.g. `'layer "layer_name" (LayerClass)'`.

    Returns:
      A wrapped version of `fn`.
    """

    def error_handler(*args, **kwargs):
        signature = None
        bound_signature = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if hasattr(e, '_keras_call_info_injected'):
                raise e
            signature = inspect.signature(fn)
            try:
                bound_signature = signature.bind(*args, **kwargs)
            except TypeError:
                raise e
            arguments_context = []
            for arg in list(signature.parameters.values()):
                if arg.name in bound_signature.arguments:
                    value = tf.nest.map_structure(format_argument_value, bound_signature.arguments[arg.name])
                else:
                    value = arg.default
                arguments_context.append(f'  • {arg.name}={value}')
            if arguments_context:
                arguments_context = '\n'.join(arguments_context)
                if isinstance(e, tf.errors.OpError):
                    message = e.message
                elif e.args:
                    message = e.args[0]
                else:
                    message = ''
                display_name = f'{(object_name if object_name else fn.__name__)}'
                message = f'Exception encountered when calling {display_name}.\n\n{message}\n\nCall arguments received by {display_name}:\n{arguments_context}'
                if isinstance(e, tf.errors.OpError):
                    new_e = e.__class__(e.node_def, e.op, message, e.error_code)
                else:
                    try:
                        new_e = e.__class__(message)
                    except TypeError:
                        new_e = RuntimeError(message)
                new_e._keras_call_info_injected = True
            else:
                new_e = e
            raise new_e.with_traceback(e.__traceback__) from None
        finally:
            del signature
            del bound_signature
    return tf.__internal__.decorator.make_decorator(fn, error_handler)
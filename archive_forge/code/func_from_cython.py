from . import _ccallback_c
import ctypes
@classmethod
def from_cython(cls, module, name, user_data=None, signature=None):
    """
        Create a low-level callback function from an exported Cython function.

        Parameters
        ----------
        module : module
            Cython module where the exported function resides
        name : str
            Name of the exported function
        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
            User data to pass on to the callback function.
        signature : str, optional
            Signature of the function. If omitted, determined from *function*.

        """
    try:
        function = module.__pyx_capi__[name]
    except AttributeError as e:
        message = 'Given module is not a Cython module with __pyx_capi__ attribute'
        raise ValueError(message) from e
    except KeyError as e:
        message = f'No function {name!r} found in __pyx_capi__ of the module'
        raise ValueError(message) from e
    return cls(function, user_data, signature)
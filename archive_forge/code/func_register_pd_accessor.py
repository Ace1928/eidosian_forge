from types import ModuleType
from typing import Any, Union
import modin.pandas as pd
def register_pd_accessor(name: str):
    """
    Registers a pd namespace attribute with the name provided.

    This is a decorator that assigns a new attribute to modin.pandas. It can be used
    with the following syntax:

    ```
    @register_pd_accessor("new_function")
    def my_new_pd_function(*args, **kwargs):
        # logic goes here
        return
    ```

    The new attribute can then be accessed with the name provided:

    ```
    import modin.pandas as pd

    pd.new_method(*my_args, **my_kwargs)
    ```


    Parameters
    ----------
    name : str
        The name of the attribute to assign to modin.pandas.

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    return _set_attribute_on_obj(name, pd._PD_EXTENSIONS_, pd)
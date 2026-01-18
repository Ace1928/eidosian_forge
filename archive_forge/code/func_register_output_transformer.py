import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, no_type_check
from triad import ParamDict, Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import is_class_method, parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
from fugue.extensions.transformer.transformer import CoTransformer, Transformer
from .._utils import (
def register_output_transformer(alias: str, obj: Any, on_dup: int=ParamDict.OVERWRITE) -> None:
    """Register output transformer with an alias.

    :param alias: alias of the transformer
    :param obj: the object that can be converted to
        :class:`~fugue.extensions.transformer.transformer.OutputTransformer` or
        :class:`~fugue.extensions.transformer.transformer.OutputCoTransformer`
    :param on_dup: see :meth:`triad.collections.dict.ParamDict.update`
        , defaults to ``ParamDict.OVERWRITE``

    .. tip::

        Registering an extension with an alias is particularly useful for projects
        such as libraries. This is because by using alias, users don't have to
        import the specific extension, or provide the full path of the extension.
        It can make user's code less dependent and easy to understand.

    .. admonition:: New Since
        :class: hint

        **0.6.0**

    .. seealso::

        Please read |TransformerTutorial|

    .. admonition:: Examples

        Here is an example how you setup your project so your users can
        benefit from this feature. Assume your project name is ``pn``

        The transformer implementation in file ``pn/pn/transformers.py``

        .. code-block:: python

            import pandas as pd

            def my_transformer(df:pd.DataFrame) -> None:
                df.to_parquet("<unique_path>")

        Then in ``pn/pn/__init__.py``

        .. code-block:: python

            from .transformers import my_transformer
            from fugue import register_transformer

            def register_extensions():
                register_transformer("mt", my_transformer)
                # ... register more extensions

            register_extensions()

        In users code:

        .. code-block:: python

            import pn  # register_extensions will be called
            from fugue import FugueWorkflow

            dag = FugueWorkflow()
            # use my_transformer by alias
            dag.df([[0]],"a:int").out_transform("mt")
            dag.run()
    """
    _OUT_TRANSFORMER_REGISTRY.update({alias: obj}, on_dup=on_dup)
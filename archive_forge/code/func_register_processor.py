import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame, DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.processor.processor import Processor
from .._utils import (
def register_processor(alias: str, obj: Any, on_dup: int=ParamDict.OVERWRITE) -> None:
    """Register processor with an alias.

    :param alias: alias of the processor
    :param obj: the object that can be converted to
        :class:`~fugue.extensions.processor.processor.Processor`
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

        Please read
        :doc:`Processor Tutorial <tutorial:tutorials/extensions/processor>`

    .. admonition:: Examples

        Here is an example how you setup your project so your users can
        benefit from this feature. Assume your project name is ``pn``

        The processor implementation in file ``pn/pn/processors.py``

        .. code-block:: python

            from fugue import DataFrame

            def my_processor(df:DataFrame) -> DataFrame:
                return df

        Then in ``pn/pn/__init__.py``

        .. code-block:: python

            from .processors import my_processor
            from fugue import register_processor

            def register_extensions():
                register_processor("mp", my_processor)
                # ... register more extensions

            register_extensions()

        In users code:

        .. code-block:: python

            import pn  # register_extensions will be called
            from fugue import FugueWorkflow

            dag = FugueWorkflow()
            # use my_processor by alias
            dag.df([[0]],"a:int").process("mp").show()
            dag.run()
    """
    _PROCESSOR_REGISTRY.update({alias: obj}, on_dup=on_dup)
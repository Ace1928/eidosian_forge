import json
import os
from docker import errors
from docker.context.config import get_meta_dir
from docker.context.config import METAFILE
from docker.context.config import get_current_context_name
from docker.context.config import write_context_name_to_docker_config
from docker.context import Context
@classmethod
def inspect_context(cls, name='default'):
    """Remove a context. Similar to the ``docker context inspect`` command.

        Args:
            name (str): The name of the context

        Raises:
            :py:class:`docker.errors.MissingContextParameter`
                If a context name is not provided.
            :py:class:`docker.errors.ContextNotFound`
                If a context with the name does not exist.

        Example:

        >>> from docker.context import ContextAPI
        >>> ContextAPI.remove_context(name='test')
        >>>
        """
    if not name:
        raise errors.MissingContextParameter('name')
    if name == 'default':
        return cls.DEFAULT_CONTEXT()
    ctx = Context.load_context(name)
    if not ctx:
        raise errors.ContextNotFound(name)
    return ctx()
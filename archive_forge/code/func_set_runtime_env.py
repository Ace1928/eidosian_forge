import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def set_runtime_env(self, runtime_env: Optional[Union[Dict[str, Any], 'RuntimeEnv']], validate: bool=False) -> None:
    """Modify the runtime_env of the JobConfig.

        We don't validate the runtime_env by default here because it may go
        through some translation before actually being passed to C++ (e.g.,
        working_dir translated from a local directory to a URI).

        Args:
            runtime_env: A :ref:`runtime environment <runtime-environments>` dictionary.
            validate: Whether to validate the runtime env.
        """
    self.runtime_env = runtime_env if runtime_env is not None else {}
    if validate:
        self.runtime_env = self._validate_runtime_env()
    self._cached_pb = None
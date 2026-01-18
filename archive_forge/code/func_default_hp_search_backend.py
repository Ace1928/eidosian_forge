from .integrations import (
from .trainer_utils import (
from .utils import logging
def default_hp_search_backend() -> str:
    available_backends = [backend for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values() if backend.is_available()]
    if len(available_backends) > 0:
        name = available_backends[0].name
        if len(available_backends) > 1:
            logger.info(f'{len(available_backends)} hyperparameter search backends available. Using {name} as the default.')
        return name
    raise RuntimeError('No hyperparameter search backend available.\n' + '\n'.join((f' - To install {backend.name} run {backend.pip_install()}' for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values())))
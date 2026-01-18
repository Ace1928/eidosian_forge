import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def env_render_passive_checker(env, *args, **kwargs):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared."""
    render_modes = env.metadata.get('render_modes')
    if render_modes is None:
        logger.warn("No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`.")
    else:
        if not isinstance(render_modes, (list, tuple)):
            logger.warn(f'Expects the render_modes to be a sequence (i.e. list, tuple), actual type: {type(render_modes)}')
        elif not all((isinstance(mode, str) for mode in render_modes)):
            logger.warn(f'Expects all render modes to be strings, actual types: {[type(mode) for mode in render_modes]}')
        render_fps = env.metadata.get('render_fps')
        if len(render_modes) > 0:
            if render_fps is None:
                logger.warn("No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.")
            elif not (np.issubdtype(type(render_fps), np.integer) or np.issubdtype(type(render_fps), np.floating)):
                logger.warn(f"Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: {type(render_fps)}")
            else:
                assert render_fps > 0, f"Expects the `env.metadata['render_fps']` to be greater than zero, actual value: {render_fps}"
        if len(render_modes) == 0:
            assert env.render_mode is None, f'With no render_modes, expects the Env.render_mode to be None, actual value: {env.render_mode}'
        else:
            assert env.render_mode is None or env.render_mode in render_modes, f'The environment was initialized successfully however with an unsupported render mode. Render mode: {env.render_mode}, modes: {render_modes}'
    result = env.render(*args, **kwargs)
    return result
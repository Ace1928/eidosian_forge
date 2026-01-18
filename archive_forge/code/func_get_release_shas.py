import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_release_shas() -> Dict[str, str]:
    common_release_envs = ['VERCEL_GIT_COMMIT_SHA', 'NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA', 'COMMIT_REF', 'RENDER_GIT_COMMIT', 'CI_COMMIT_SHA', 'CIRCLE_SHA1', 'CF_PAGES_COMMIT_SHA', 'REACT_APP_GIT_SHA', 'SOURCE_VERSION', 'GITHUB_SHA', 'TRAVIS_COMMIT', 'GIT_COMMIT', 'BUILD_VCS_NUMBER', 'bamboo_planRepository_revision', 'Build.SourceVersion', 'BITBUCKET_COMMIT', 'DRONE_COMMIT_SHA', 'SEMAPHORE_GIT_SHA', 'BUILDKITE_COMMIT']
    shas = {}
    for env in common_release_envs:
        env_var = os.environ.get(env)
        if env_var is not None:
            shas[env] = env_var
    return shas
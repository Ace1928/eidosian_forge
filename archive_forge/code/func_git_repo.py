from pprint import pformat
from six import iteritems
import re
@git_repo.setter
def git_repo(self, git_repo):
    """
        Sets the git_repo of this V1Volume.
        GitRepo represents a git repository at a particular revision.
        DEPRECATED: GitRepo is deprecated. To provision a container with a git
        repo, mount an EmptyDir into an InitContainer that clones the repo using
        git, then mount the EmptyDir into the Pod's container.

        :param git_repo: The git_repo of this V1Volume.
        :type: V1GitRepoVolumeSource
        """
    self._git_repo = git_repo
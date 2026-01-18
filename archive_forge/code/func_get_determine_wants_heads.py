import itertools
from typing import Callable, Dict, Tuple, Optional
from dulwich.errors import NotCommitError
from dulwich.objects import ObjectID
from dulwich.object_store import ObjectStoreGraphWalker
from dulwich.pack import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.protocol import CAPABILITY_THIN_PACK, ZERO_SHA
from dulwich.refs import SYMREF
from dulwich.walk import Walker
from .. import config, trace, ui
from ..errors import (DivergedBranches, FetchLimitUnsupported,
from ..repository import FetchResult, InterRepository, AbstractSearchResult
from ..revision import NULL_REVISION, RevisionID
from .errors import NoPushSupport
from .fetch import DetermineWantsRecorder, import_git_objects
from .mapping import needs_roundtripping
from .object_store import get_object_store
from .push import MissingObjectsIterator, remote_divergence
from .refs import is_tag, ref_to_tag_name
from .remote import RemoteGitError, RemoteGitRepository
from .repository import GitRepository, GitRepositoryFormat, LocalGitRepository
from .unpeel_map import UnpeelMap
def get_determine_wants_heads(self, wants, include_tags=False, tag_selector=None):
    wants = set(wants)

    def determine_wants(refs):
        unpeel_lookup = {}
        for k, v in refs.items():
            if k.endswith(PEELED_TAG_SUFFIX):
                unpeel_lookup[v] = refs[k[:-len(PEELED_TAG_SUFFIX)]]
        potential = {unpeel_lookup.get(w, w) for w in wants}
        if include_tags:
            for k, sha in refs.items():
                if k.endswith(PEELED_TAG_SUFFIX):
                    continue
                try:
                    tag_name = ref_to_tag_name(k)
                except ValueError:
                    continue
                if tag_selector and (not tag_selector(tag_name)):
                    continue
                if sha == ZERO_SHA:
                    continue
                potential.add(sha)
        return list(potential - self._target_has_shas(potential))
    return determine_wants
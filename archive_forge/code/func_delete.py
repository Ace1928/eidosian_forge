import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def delete(cls, repo: 'Repo', path: PathLike) -> None:
    """Delete the reference at the given path.

        :param repo:
            Repository to delete the reference from.

        :param path:
            Short or full path pointing to the reference, e.g. ``refs/myreference``
            or just ``myreference``, hence ``refs/`` is implied.
            Alternatively the symbolic reference to be deleted.
        """
    full_ref_path = cls.to_full_path(path)
    abs_path = os.path.join(repo.common_dir, full_ref_path)
    if os.path.exists(abs_path):
        os.remove(abs_path)
    else:
        pack_file_path = cls._get_packed_refs_path(repo)
        try:
            with open(pack_file_path, 'rb') as reader:
                new_lines = []
                made_change = False
                dropped_last_line = False
                for line_bytes in reader:
                    line = line_bytes.decode(defenc)
                    _, _, line_ref = line.partition(' ')
                    line_ref = line_ref.strip()
                    if (line.startswith('#') or full_ref_path != line_ref) and (not dropped_last_line or (dropped_last_line and (not line.startswith('^')))):
                        new_lines.append(line)
                        dropped_last_line = False
                        continue
                    made_change = True
                    dropped_last_line = True
            if made_change:
                with open(pack_file_path, 'wb') as fd:
                    fd.writelines((line.encode(defenc) for line in new_lines))
        except OSError:
            pass
    reflog_path = RefLog.path(cls(repo, full_ref_path))
    if os.path.isfile(reflog_path):
        os.remove(reflog_path)
import stat
from base64 import standard_b64decode
from dulwich.objects import Blob, Tree
def store_git_pristine_tar_data(repo, filename, delta, gitid, message=None, **kwargs):
    """Add pristine tar data to a Git repository.

    :param repo: Git repository to add data to
    :param filename: Name of file to store for
    :param delta: pristine-tar delta
    :param gitid: Git id the pristine tar delta is generated against
    """
    delta_ob = Blob.from_string(delta)
    delta_name = filename + b'.delta'
    id_ob = Blob.from_string(gitid)
    id_name = filename + b'.id'
    objects = [(delta_ob, delta_name), (id_ob, id_name)]
    tree = get_pristine_tar_tree(repo)
    tree.add(delta_name, stat.S_IFREG | 420, delta_ob.id)
    tree.add(id_name, stat.S_IFREG | 420, id_ob.id)
    if b'README' not in tree:
        readme_ob = Blob.from_string(README_CONTENTS)
        objects.append((readme_ob, b'README'))
        tree.add(b'README', stat.S_IFREG | 420, readme_ob.id)
    objects.append((tree, ''))
    repo.object_store.add_objects(objects)
    if message is None:
        message = b'pristine-tar data for %s' % filename
    return repo.do_commit(ref=b'refs/heads/pristine-tar', tree=tree.id, message=message, **kwargs)
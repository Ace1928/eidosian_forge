import re
def git_to_bzr(self, ref_name):
    """Map a git reference name to a Bazaar branch name.
        """
    parts = ref_name.split(b'/')
    if parts[0] == b'refs':
        parts.pop(0)
    category = parts.pop(0)
    if category == b'heads':
        git_name = b'/'.join(parts)
        bazaar_name = self._git_to_bzr_name(git_name)
    else:
        if category == b'remotes' and parts[0] == b'origin':
            parts.pop(0)
        git_name = b'/'.join(parts)
        if category.endswith(b's'):
            category = category[:-1]
        name_no_ext = self._git_to_bzr_name(git_name)
        bazaar_name = '{}.{}'.format(name_no_ext, category.decode('ascii'))
    return bazaar_name
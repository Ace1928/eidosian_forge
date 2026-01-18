from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def do_diff(file_id, old_path, new_path, action, force_binary):

    def tree_lines(tree, path, require_text=False):
        try:
            tree_file = tree.get_file(path)
        except _mod_transport.NoSuchFile:
            return []
        else:
            if require_text is True:
                tree_file = text_file(tree_file)
            return tree_file.readlines()
    try:
        if force_binary:
            raise errors.BinaryFile()
        old_lines = tree_lines(old_tree, old_path, require_text=True)
        new_lines = tree_lines(new_tree, new_path, require_text=True)
        action.write(self.to_file)
        internal_diff(old_path, old_lines, new_path, new_lines, self.to_file)
    except errors.BinaryFile:
        old_lines = tree_lines(old_tree, old_path, require_text=False)
        new_lines = tree_lines(new_tree, new_path, require_text=False)
        action.add_property('encoding', 'base64')
        action.write(self.to_file)
        binary_diff(old_path, old_lines, new_path, new_lines, self.to_file)
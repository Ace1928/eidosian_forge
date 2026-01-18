from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def report_delta(to_file, delta, short_status=False, show_ids=False, show_unchanged=False, indent='', predicate=None, classify=True):
    """Output this delta in status-like form to to_file.

    :param to_file: A file-like object where the output is displayed.

    :param delta: A TreeDelta containing the changes to be displayed

    :param short_status: Single-line status if True.

    :param show_ids: Output the file ids if True.

    :param show_unchanged: Output the unchanged files if True.

    :param indent: Added at the beginning of all output lines (for merged
        revisions).

    :param predicate: A callable receiving a path returning True if the path
        should be displayed.

    :param classify: Add special symbols to indicate file kind.
    """

    def decorate_path(path, kind, meta_modified=None):
        if not classify:
            return path
        if kind == 'directory':
            path += '/'
        elif kind == 'symlink':
            path += '@'
        if meta_modified:
            path += '*'
        return path

    def show_more_renamed(item):
        dec_new_path = decorate_path(item.path[1], item.kind[1], item.meta_modified())
        to_file.write(' => %s' % dec_new_path)
        if item.changed_content or item.meta_modified():
            extra_modified.append(InventoryTreeChange(item.file_id, (item.path[1], item.path[1]), item.changed_content, item.versioned, (item.parent_id[1], item.parent_id[1]), (item.name[1], item.name[1]), (item.kind[1], item.kind[1]), item.executable))

    def show_more_kind_changed(item):
        to_file.write(' ({} => {})'.format(item.kind[0], item.kind[1]))

    def show_path(path, kind, meta_modified, default_format, with_file_id_format):
        dec_path = decorate_path(path, kind, meta_modified)
        if show_ids:
            to_file.write(with_file_id_format % dec_path)
        else:
            to_file.write(default_format % dec_path)

    def show_list(files, long_status_name, short_status_letter, default_format='%s', with_file_id_format='%-30s', show_more=None):
        if files:
            header_shown = False
            if short_status:
                prefix = short_status_letter
            else:
                prefix = ''
            prefix = indent + prefix + '  '
            for item in files:
                if item.path[0] is None:
                    path = item.path[1]
                    kind = item.kind[1]
                else:
                    path = item.path[0]
                    kind = item.kind[0]
                if predicate is not None and (not predicate(path)):
                    continue
                if not header_shown and (not short_status):
                    to_file.write(indent + long_status_name + ':\n')
                    header_shown = True
                to_file.write(prefix)
                show_path(path, kind, item.meta_modified(), default_format, with_file_id_format)
                if show_more is not None:
                    show_more(item)
                if show_ids and getattr(item, 'file_id', None):
                    to_file.write(' %s' % item.file_id.decode('utf-8'))
                to_file.write('\n')
    show_list(delta.removed, 'removed', 'D')
    show_list(delta.added, 'added', 'A')
    show_list(delta.missing, 'missing', '!')
    extra_modified = []
    show_list(delta.renamed, 'renamed', 'R', with_file_id_format='%s', show_more=show_more_renamed)
    show_list(delta.copied, 'copied', 'C', with_file_id_format='%s', show_more=show_more_renamed)
    show_list(delta.kind_changed, 'kind changed', 'K', with_file_id_format='%s', show_more=show_more_kind_changed)
    show_list(delta.modified + extra_modified, 'modified', 'M')
    if show_unchanged:
        show_list(delta.unchanged, 'unchanged', 'S')
    show_list(delta.unversioned, 'unknown', ' ')
import re
def task_lists(md):
    """A mistune plugin to support task lists. Spec defined by
    GitHub flavored Markdown and commonly used by many parsers:

    .. code-block:: text

        - [ ] unchecked task
        - [x] checked task

    :param md: Markdown instance
    """
    md.before_render_hooks.append(task_lists_hook)
    if md.renderer and md.renderer.NAME == 'html':
        md.renderer.register('task_list_item', render_task_list_item)
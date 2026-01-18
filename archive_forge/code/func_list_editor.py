import datetime
from functools import partial
import logging
def list_editor(trait, handler):
    """ Factory that constructs an appropriate editor for a list.
    """
    item_handler = handler.item_trait.handler
    if _expects_hastraits_instance(item_handler):
        from traitsui.table_filter import EvalFilterTemplate, RuleFilterTemplate, MenuFilterTemplate, EvalTableFilter
        from traitsui.api import TableEditor
        return TableEditor(filters=[RuleFilterTemplate, MenuFilterTemplate, EvalFilterTemplate], edit_view='', orientation='vertical', search=EvalTableFilter(), deletable=True, show_toolbar=True, reorderable=True, row_factory=_instance_handler_factory(item_handler))
    else:
        from traitsui.api import ListEditor
        return ListEditor(trait_handler=handler, rows=trait.rows if trait.rows else 5, use_notebook=bool(trait.use_notebook), page_name=trait.page_name if trait.page_name else '')
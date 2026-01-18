import re
def render_task_list_item(renderer, text, checked=False):
    checkbox = '<input class="task-list-item-checkbox" type="checkbox" disabled'
    if checked:
        checkbox += ' checked/>'
    else:
        checkbox += '/>'
    if text.startswith('<p>'):
        text = text.replace('<p>', '<p>' + checkbox, 1)
    else:
        text = checkbox + text
    return '<li class="task-list-item">' + text + '</li>\n'
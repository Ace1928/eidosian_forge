import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def make_task(self, parent_task, view, msg, curr, total):
    task = ProgressTask(parent_task, progress_view=view)
    task.msg = msg
    task.current_cnt = curr
    task.total_cnt = total
    return task
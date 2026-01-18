import warnings
def nested_progress_bar(self):
    """Return a nested progress bar.

        When the bar has been finished with, it should be released by calling
        bar.finished().
        """
    from ..progress import ProgressTask
    if self._task_stack:
        t = ProgressTask(self._task_stack[-1], self)
    else:
        t = ProgressTask(None, self)
    self._task_stack.append(t)
    return t
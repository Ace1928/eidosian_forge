import os
import time
def make_sub_task(self):
    return ProgressTask(self, ui_factory=self.ui_factory, progress_view=self.progress_view)
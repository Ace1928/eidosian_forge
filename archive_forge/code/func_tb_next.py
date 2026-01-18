import functools
import traceback
@property
def tb_next(self):
    tb_next = self._tb.tb_next
    if tb_next and tb_next != self._filtered_traceback:
        return FilteredTraceback(tb_next, self._filtered_traceback)
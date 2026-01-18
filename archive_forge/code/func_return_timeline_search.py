from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
def return_timeline_search(self, results: Dict=None):
    if not results:
        return None
    if self._output_format.value == 'dict':
        return results
    if self._output_format.value == 'pd':
        formatted = pd.DataFrame(results)
        formatted['datetime'] = pd.to_datetime(formatted['datetime'])
        return formatted
    if self._output_format.value == 'json':
        return LazyJson.dumps(results)
    if self._output_format.value == 'obj':
        return [LazyObject(res) for res in results]
from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
def article_search(self, filters: GDELTFilters) -> Union[pd.DataFrame, Dict, str]:
    articles = self._query('artlist', filters.query_string)
    return self.return_article_result(articles)
from ._base import *
from .filters import GDELTFilters
from .models import GDELTArticle
def return_article_result(self, articles: Dict=None):
    if not articles or not articles.get('articles'):
        return None
    if self._output_format.value == 'dict':
        return articles['articles']
    if self._output_format.value == 'pd':
        return pd.DataFrame(articles['articles'])
    if self._output_format.value == 'json':
        return LazyJson.dumps(articles['articles'])
    if self._output_format.value == 'obj':
        return [GDELTArticle(**article) for article in articles['articles']]
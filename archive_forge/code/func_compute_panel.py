from .._utils import groupby_apply
from ..doctools import document
from ..mapping.aes import ALL_AESTHETICS
from ..mapping.evaluation import after_stat
from .stat import stat
@classmethod
def compute_panel(cls, data, scales, **params):
    if 'weight' not in data:
        data['weight'] = 1

    def count(df):
        """
            Do a weighted count
            """
        df['n'] = df['weight'].sum()
        return df.iloc[0:1]

    def ave(df):
        """
            Calculate proportion values
            """
        df['prop'] = df['n'] / df['n'].sum()
        return df
    s: set[str] = set(data.columns) & ALL_AESTHETICS
    by = list(s.difference(['weight']))
    counts = groupby_apply(data, by, count)
    counts = groupby_apply(counts, 'group', ave)
    return counts
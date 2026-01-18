from typing import TYPE_CHECKING, List
def concat_and_sort(blocks: List['pyarrow.Table'], sort_key: 'SortKey') -> 'pyarrow.Table':
    check_polars_installed()
    blocks = [pl.from_arrow(block) for block in blocks]
    df = pl.concat(blocks).sort(sort_key.get_columns(), reverse=sort_key.get_descending())
    return df.to_arrow()
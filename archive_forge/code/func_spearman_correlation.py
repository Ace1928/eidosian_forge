def spearman_correlation(ranks1, ranks2):
    """Returns the Spearman correlation coefficient for two rankings, which
    should be dicts or sequences of (key, rank). The coefficient ranges from
    -1.0 (ranks are opposite) to 1.0 (ranks are identical), and is only
    calculated for keys in both rankings (for meaningful results, remove keys
    present in only one list before ranking)."""
    n = 0
    res = 0
    for k, d in _rank_dists(ranks1, ranks2):
        res += d * d
        n += 1
    try:
        return 1 - 6 * res / (n * (n * n - 1))
    except ZeroDivisionError:
        return 0.0
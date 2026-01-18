def iris():
    """
    Each row represents a flower.

    https://en.wikipedia.org/wiki/Iris_flower_data_set

    Returns:
        A `pandas.DataFrame` with 150 rows and the following columns:
        `['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'species_id']`."""
    return _get_dataset('iris')
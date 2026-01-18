def gapminder(datetimes=False, centroids=False, year=None, pretty_names=False):
    """
    Each row represents a country on a given year.

    https://www.gapminder.org/data/

    Returns:
        A `pandas.DataFrame` with 1704 rows and the following columns:
        `['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap',
        'iso_alpha', 'iso_num']`.
        If `datetimes` is True, the 'year' column will be a datetime column
        If `centroids` is True, two new columns are added: ['centroid_lat', 'centroid_lon']
        If `year` is an integer, the dataset will be filtered for that year
    """
    df = _get_dataset('gapminder')
    if year:
        df = df[df['year'] == year]
    if datetimes:
        df['year'] = (df['year'].astype(str) + '-01-01').astype('datetime64[ns]')
    if not centroids:
        df = df.drop(['centroid_lat', 'centroid_lon'], axis=1)
    if pretty_names:
        df.rename(mapper=dict(country='Country', continent='Continent', year='Year', lifeExp='Life Expectancy', gdpPercap='GDP per Capita', pop='Population', iso_alpha='ISO Alpha Country Code', iso_num='ISO Numeric Country Code', centroid_lat='Centroid Latitude', centroid_lon='Centroid Longitude'), axis='columns', inplace=True)
    return df
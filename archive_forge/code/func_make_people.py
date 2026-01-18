from __future__ import annotations
import random
from packaging.version import Version
from dask.utils import import_required
def make_people(npartitions=10, records_per_partition=1000, seed=None, locale='en'):
    """Make a dataset of random people

    This makes a Dask Bag with dictionary records of randomly generated people.
    This requires the optional library ``mimesis`` to generate records.

    Parameters
    ----------
    npartitions : int
        Number of partitions
    records_per_partition : int
        Number of records in each partition
    seed : int, (optional)
        Random seed
    locale : str
        Language locale, like 'en', 'fr', 'zh', or 'ru'

    Returns
    -------
    b: Dask Bag
    """
    import_required('mimesis', 'The mimesis module is required for this function.  Try:\n  python -m pip install mimesis')
    schema = lambda field: {'age': field('random.randint', a=0, b=120), 'name': (field('person.name'), field('person.surname')), 'occupation': field('person.occupation'), 'telephone': field('person.telephone'), 'address': {'address': field('address.address'), 'city': field('address.city')}, 'credit-card': {'number': field('payment.credit_card_number'), 'expiration-date': field('payment.credit_card_expiration_date')}}
    return _make_mimesis({'locale': locale}, schema, npartitions, records_per_partition, seed)
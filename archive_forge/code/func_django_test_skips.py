from django.contrib.gis.db.backends.base.features import BaseSpatialFeatures
from django.db.backends.sqlite3.features import (
from django.utils.functional import cached_property
@cached_property
def django_test_skips(self):
    skips = super().django_test_skips
    skips.update({"SpatiaLite doesn't support distance lookups with Distance objects.": {'gis_tests.geogapp.tests.GeographyTest.test02_distance_lookup'}})
    return skips
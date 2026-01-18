import testtools
import yaql
from yaql.language import factory
from yaql import legacy
def create_legacy_engine(self):
    func = TestCase._default_legacy_engine
    if func is None:
        engine_factory = legacy.YaqlFactory()
        TestCase._default_legacy_engine = func = engine_factory.create(options=self.legacy_engine_options)
    return func
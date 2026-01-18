from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_topics(self):
    """Smoketest for 'brz help topics'"""
    out, err = self.run_bzr('help topics')
    self.assertContainsRe(out, 'basic')
    self.assertContainsRe(out, 'topics')
    self.assertContainsRe(out, 'commands')
    self.assertContainsRe(out, 'revisionspec')
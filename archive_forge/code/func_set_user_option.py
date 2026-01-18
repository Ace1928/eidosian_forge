from .. import config
def set_user_option(self, name, value, store=config.STORE_BRANCH, warn_masked=False):
    """Force local to True"""
    config.BranchConfig.set_user_option(self, name, value, store=config.STORE_LOCATION, warn_masked=warn_masked)
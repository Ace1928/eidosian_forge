from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@property
def marketplace(self):
    try:
        return self._marketplace
    except AttributeError:
        response = self.mws.list_marketplace_participations()
        result = response.ListMarketplaceParticipationsResult
        self._marketplace = result.ListMarketplaces.Marketplace[0]
        return self.marketplace
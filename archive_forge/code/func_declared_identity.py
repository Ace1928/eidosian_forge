from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def declared_identity(self, ctx, declared):
    return _NoOne()
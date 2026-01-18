from pprint import pformat
from six import iteritems
import re
@conversion_review_versions.setter
def conversion_review_versions(self, conversion_review_versions):
    """
        Sets the conversion_review_versions of this
        V1beta1CustomResourceConversion.
        ConversionReviewVersions is an ordered list of preferred
        `ConversionReview` versions the Webhook expects. API server will try to
        use first version in the list which it supports. If none of the versions
        specified in this list supported by API server, conversion will fail for
        this object. If a persisted Webhook configuration specifies allowed
        versions and does not include any versions known to the API Server,
        calls to the webhook will fail. Default to `['v1beta1']`.

        :param conversion_review_versions: The conversion_review_versions of
        this V1beta1CustomResourceConversion.
        :type: list[str]
        """
    self._conversion_review_versions = conversion_review_versions
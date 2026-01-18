from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def list_jobs_by_status(self, status=None, ascending=None, page_token=None):
    """
        The ListJobsByStatus operation gets a list of jobs that have a
        specified status. The response body contains one element for
        each job that satisfies the search criteria.

        :type status: string
        :param status: To get information about all of the jobs associated with
            the current AWS account that have a given status, specify the
            following status: `Submitted`, `Progressing`, `Complete`,
            `Canceled`, or `Error`.

        :type ascending: string
        :param ascending: To list jobs in chronological order by the date and
            time that they were submitted, enter `True`. To list jobs in
            reverse chronological order, enter `False`.

        :type page_token: string
        :param page_token: When Elastic Transcoder returns more than one page
            of results, use `pageToken` in subsequent `GET` requests to get
            each successive page of results.

        """
    uri = '/2012-09-25/jobsByStatus/{0}'.format(status)
    params = {}
    if status is not None:
        params['Status'] = status
    if ascending is not None:
        params['Ascending'] = ascending
    if page_token is not None:
        params['PageToken'] = page_token
    return self.make_request('GET', uri, expected_status=200, params=params)
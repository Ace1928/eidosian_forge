import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def get_all_qualifications_for_qual_type(self, qualification_type_id):
    page_size = 100
    search_qual = self.get_qualifications_for_qualification_type(qualification_type_id)
    total_records = int(search_qual.TotalNumResults)
    get_page_quals = lambda page: self.get_qualifications_for_qualification_type(qualification_type_id=qualification_type_id, page_size=page_size, page_number=page)
    page_nums = self._get_pages(page_size, total_records)
    qual_sets = itertools.imap(get_page_quals, page_nums)
    return itertools.chain.from_iterable(qual_sets)
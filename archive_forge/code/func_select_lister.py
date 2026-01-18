from boto.compat import six
def select_lister(domain, query='', max_items=None):
    more_results = True
    num_results = 0
    next_token = None
    while more_results:
        rs = domain.connection.select(domain, query, next_token=next_token)
        for item in rs:
            if max_items:
                if num_results == max_items:
                    raise StopIteration
            yield item
            num_results += 1
        next_token = rs.next_token
        more_results = next_token is not None
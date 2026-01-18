import logging
def put_item(self, Item):
    self._add_request_and_process({'PutRequest': {'Item': Item}})
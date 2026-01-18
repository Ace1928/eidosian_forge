import logging
def delete_item(self, Key):
    self._add_request_and_process({'DeleteRequest': {'Key': Key}})
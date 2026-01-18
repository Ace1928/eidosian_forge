from bs4 import BeautifulSoup
def clear_session_storage(self):
    self.driver.execute_script('window.sessionStorage.clear()')
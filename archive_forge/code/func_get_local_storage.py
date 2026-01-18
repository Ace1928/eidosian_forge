from bs4 import BeautifulSoup
def get_local_storage(self, store_id='local'):
    return self.driver.execute_script(f"return JSON.parse(window.localStorage.getItem('{store_id}'));")
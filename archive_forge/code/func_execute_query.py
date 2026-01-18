import pandas as pd
import sqlite3
import math
import numpy as np
from typing import List, Dict, Any
import yfinance as yf
import yfinance as yf
import pandas as pd
import sqlite3
def execute_query(self, query: str, params: tuple=()):
    """Execute a single SQL query."""
    self.cursor.execute(query, params)
    self.conn.commit()
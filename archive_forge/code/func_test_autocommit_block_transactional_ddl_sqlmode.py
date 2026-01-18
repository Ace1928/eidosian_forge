import io
from ...migration import MigrationContext
from ...testing import assert_raises
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import is_false
from ...testing import is_not_
from ...testing import is_true
from ...testing import ne_
from ...testing.fixtures import TestBase
def test_autocommit_block_transactional_ddl_sqlmode(self):
    context = self._fixture({'transaction_per_migration': True, 'transactional_ddl': True, 'as_sql': True})
    with context.begin_transaction():
        context.execute('step 1')
        with context.begin_transaction(_per_migration=True):
            context.execute('step 2')
            with context.autocommit_block():
                context.execute('step 3')
            context.execute('step 4')
        context.execute('step 5')
    self._assert_impl_steps('step 1', 'BEGIN', 'step 2', 'COMMIT', 'step 3', 'BEGIN', 'step 4', 'COMMIT', 'step 5')
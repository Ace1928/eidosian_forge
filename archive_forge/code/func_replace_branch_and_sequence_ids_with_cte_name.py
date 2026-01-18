from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.helper import ensure_list
def replace_branch_and_sequence_ids_with_cte_name(spark: SparkSession, expression_context: exp.Select, id: exp.Identifier):
    if id.alias_or_name in spark.known_ids:
        if expression_context.args.get('joins') and id.alias_or_name in spark.known_branch_ids:
            join_table_aliases = [x.alias_or_name for x in get_tables_from_expression_with_join(expression_context)]
            ctes_in_join = [cte for cte in expression_context.ctes if cte.alias_or_name in join_table_aliases]
            if ctes_in_join[0].args['branch_id'] == ctes_in_join[1].args['branch_id']:
                assert len(ctes_in_join) == 2
                _set_alias_name(id, ctes_in_join[0].alias_or_name)
                return
        for cte in reversed(expression_context.ctes):
            if id.alias_or_name in (cte.args['branch_id'], cte.args['sequence_id']):
                _set_alias_name(id, cte.alias_or_name)
                return